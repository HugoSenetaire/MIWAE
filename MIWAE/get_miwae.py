import torch
from .NetworkMiwae import ConvDecoder,ConvFeatureExtractor, dic_network
from .ReparametrizationTrick import ReparamTrickBernoulli, ReparamTrickNormal, dic_reparametrization
from .decoder import Decoder
from .encoder import Encoder
from .miwae import MIWAE_VAE
from torch.distributions import Normal
dic_prior = {
    "normal01" : Normal(0,1),
}


def get_networks(args_dict, input_channel =1):
    encoder_network = dic_network[args_dict["encoder_network"]](input_channel = input_channel,)
    decoder_network = dic_network[args_dict["decoder_network"]]


    return encoder_network, decoder_network

def get_encoder(args_dict, encoder_network = None,):
    reparam_encoder = dic_reparametrization[args_dict["encoder_reparam"]]()
    encoder = Encoder(reparam_trick=reparam_encoder, feature_extractor=encoder_network, latent_dim = args_dict["latent_dim"])

    return encoder

def get_decoder(args_dict, decoder_network = None, output_channel = 1):
    reparam_decoder = dic_reparametrization[args_dict["decoder_reparam"]]()
    decoder = Decoder(reparam_trick=reparam_decoder, decoder_network=decoder_network, latent_dim = args_dict["latent_dim"], output_channel=output_channel)
    return decoder


def get_miwae(args_dict, encoder, decoder,):
    prior = dic_prior[args_dict["prior"]]
    miwae = MIWAE_VAE(encoder, decoder, prior = prior)
    return miwae
